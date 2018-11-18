class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        for i in range(len(nums)):
            for j in range(1,len(nums)):
                if (nums[i]+nums[j]== target and i != j):
                    return [i,j]

# Accepted Answer
class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        nums_dict = {}
        for i in range(len(nums)):
            nums_dict[nums[i]] = i
        
        for i in range(len(nums)):
            difference = target - nums[i]
            if (difference in nums_dict and i != nums_dict[difference]):
                return [i,nums_dict[difference]]

# Accepted Answer
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        num1_stack = []
        num2_stack = []
        while (l1.next != None) :
            num1_stack.append(l1.val)
            l1 = l1.next
        num1_stack.append(l1.val)
        while (l2.next != None):
            num2_stack.append(l2.val)
            l2 = l2.next
        num2_stack.append(l2.val)
        num1 = ""
        num2 = ""
        while (len(num1_stack) != 0):
            num1 += str(num1_stack.pop())
        while (len(num2_stack) != 0):
            num2 += str(num2_stack.pop())
        result = str(int(num1) + int(num2))
        output = ListNode(result[-1])
        result = result[:-1]
        current = output
        while(len(result) != 0):
                result, current.next = result[:-1], ListNode(result[-1])
                current = current.next
        return output

#Accepted Answer
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        min_idx = 0
        max_idx = len(nums1) if len(nums1) < len(nums2) else len(nums2)
        length_1 = len(nums1)
        length_2 = len(nums2)
        smaller = nums1 if len(nums1) < len(nums2) else nums2
        larger = nums2 if len(nums1) < len(nums2) else nums1
        while(min_idx <= max_idx):
            i = (min_idx+max_idx)//2
            j = (length_1+length_2+1)//2 - i
            if (i>0  and j < len(larger) and smaller[i-1] > larger[j]):
                max_idx = i - 1
            elif (j>0 and len(smaller)>i and smaller[i] < larger[j-1]):
                min_idx = i + 1
            else:
                if (i==0):
                    median = larger[j-1]
                elif(j==0):
                    median = smaller[i-1]
                else:
                    median = max([smaller[i-1],larger[j-1]])
                break
        if ((length_1 + length_2)%2!=0):
            return median
        if (i == len(smaller)):
            return (median + larger[j])/2
        if (j == len(larger)):
            return (median + smaller[i])/2
        return (median + min([smaller[i],larger[j]]))/2
        
#Accepted Answer
class Solution:
    def isPalindrome(self, x):
        """
        :type x: int
        :rtype: bool
        """
        if (x < 0):
            return False
        num = []
        elem = x
        while(elem > 0):
            element = elem%10 
            num.append(element)
            elem = elem//10
        track = len(num)
        result = 0
        for i in range(len(num)):
            result += num[i]*10**(track-1)
            track -= 1
        if (x == result):
            return True
        return False
#Accepted Answer
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        negative = True if x < 0 else False
        number_str = ""
        mock_str = str(x)[1:] if negative else str(x)
        while (len(mock_str) > 0):
            number_str += mock_str[-1]
            mock_str = mock_str[:-1]
        final_value = -int(number_str) if negative else int(number_str)
        if (final_value < -2**31 or final_value > 2**31-1):
            return 0
        return final_value

#Accepted Answer
class Solution:
    def licenseKeyFormatting(self, S, K):
        """
        :type S: str
        :type K: int
        :rtype: str
        """
        separate = S.split("-")
        s_with_no_dash = ""
        for char in separate:
            s_with_no_dash += char
        new_str = ""
        while(len(s_with_no_dash) >= K):
            new_str = s_with_no_dash[-K:] + new_str
            s_with_no_dash = s_with_no_dash[:len(s_with_no_dash)-K]
            if (len(s_with_no_dash) > 0):
                new_str = "-" + new_str
        new_str = s_with_no_dash + new_str
        new_str = new_str.upper()
        return new_str
#Accepted Answer
class Solution:
    def nextClosestTime(self, time):
        """
        :type time: str
        :rtype: str
        """
        time_split = time.split(":")
        whole_str = time_split[0] + time_split[1]
        lower_limit_min = time_split[1]
        hr_list = [int(i+j) for i in whole_str for j in whole_str if int(i+j) < 24]
        min_list =  [int(i+j) for i in whole_str for j in whole_str if int(i+j) < 60]
        bigger_hr = sorted(list(set(([i for i in hr_list if i >= int(time_split[0])]))))
        smaller_hr = sorted(list(set([i for i in hr_list if i < int(time_split[0])])))
        bigger_min = sorted(list(set([i for i in min_list if i > int(time_split[1])])))
        smaller_min = sorted(list(set([i for i in min_list if i <= int(time_split[1])])))
        return_val = ""
        print(bigger_hr)
        if bigger_hr:
            if bigger_hr[0] == int(time_split[0]):
                if bigger_min:
                    return_val = str(bigger_hr[0]) + ":" + str(bigger_min[0])
                elif len(bigger_hr) > 1:
                    return_val = str(bigger_hr[1])+ ":" + str(min(min_list))
                else:
                    return_val = str(min(hr_list)) + ":" + str(min(min_list))
            print(return_val)
        else:
            return_val = str(min(hr_list)) + ":" + str(min(min_list))
        if return_val[1] == ":":
            return_val = "0" + return_val
        if return_val[-2] == ":":
            return_val = return_val[:-1] + "0" + return_val[-1]
        return return_val
#Accepted Answer
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        if (len(prices) in [0,1]):
            return 0
        max_profit = 0
        min_idx = 0
        smallest = prices[0]
        
        for i in range(1,len(prices)):
            if (prices[i] - prices[min_idx]> max_profit):
                max_profit = prices[i] - prices[min_idx]
            elif(prices[i] - prices[min_idx]<= 0):
                min_idx = i
        return max_profit
#Accepted Answer
class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        maxProfit = 0;
        for i in range(1,len(prices)): 
            if (prices[i] - prices[i - 1] > 0):
                maxProfit += prices[i] - prices[i - 1]  
        return maxProfit;
#Accepted Answer
class Solution(object):
    def nextClosestTime(self, time):
        cur = 60 * int(time[:2]) + int(time[3:])
        allowed = {int(x) for x in time if x != ':'}
        while True:
            cur = (cur + 1) % (24 * 60)
            if all(digit in allowed
                    for block in divmod(cur, 60)
                    for digit in divmod(block, 10)):
                return "{:02d}:{:02d}".format(*divmod(cur, 60))
#Accepted Answer
class Solution:
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n < 0:
            x = 1/x
            n = -n
        if n == 0:
            return 1
        half = self.myPow(x,n//2)
        if (n % 2 == 0):
            return half * half
        else:
            return half*half*x

#Accepted Answer
class Solution:
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        hash_roman = {}
        Roman = ["I", "V", "X", "L","C", "D", "M"]
        Number = [1,5,10,50,100,500,1000]
        hash_roman = {Roman[i] : Number[i] for i in range(len(Roman))}
        substract = {"I":["V","X"],"X":["L", "C"], "C":["D","M"]}
        converted = 0
        skip = False
        for i in range(len(s)):
            if skip:
                skip = False
                continue
            if s[i] in substract.keys() and i + 1 < len(s):
                if s[i+1] in substract[s[i]]:
                    converted += (hash_roman[s[i+1]] - hash_roman[s[i]])
                    skip = True
                    continue
            converted += hash_roman[s[i]]
        return converted

#Accepted Answer
# Write your MySQL query statement below
SELECT NAME AS CUSTOMERS FROM Customers
LEFT OUTER JOIN ORDERS
ON CUSTOMERS.ID = ORDERS.CUSTOMERID
WHERE ORDERS.ID IS NULL

#Accepted Answer
# Write your MySQL query statement below
SELECT EMAIL FROM PERSON 
GROUP BY EMAIL
HAVING COUNT(*) > 1

#Accepted Answer
class LRUCache:
    class node:
        val = 0
        key = 0
        prev = None
        nextN = None
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.collection = {}
        self.size = capacity
        self.end = None
        self.start = None
        

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if (key in self.collection.keys()):
            Node = self.collection[key]
            self.remove(Node)
            self.add(Node)
            return Node.val
        return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if (key in self.collection.keys()):
            Node = self.collection[key]
            Node.val = value
            self.remove(Node)
            self.add(Node)
        else:
            new_node = self.node()
            new_node.prev = None
            new_node.nextN = None
            new_node.val = value
            new_node.key = key
            if (len(self.collection) == self.size):
                self.collection.pop(self.end.key)
                self.remove(self.end)
                self.add(new_node)
            else:
                self.add(new_node)
            print(len(self.collection), self.size)
            self.collection[key] = new_node
            
    def remove(self,Node):
        if (Node.prev != None):
            Node.prev.nextN = Node.nextN
        else:
            self.start = Node.nextN
        
        if (Node.nextN != None):
            Node.nextN.prev = Node.prev
        else:
            self.end = Node.prev
            
        
    def add(self, Node):
        Node.prev = None
        Node.nextN = self.start
        if (self.start != None):
            self.start.prev = Node
        self.start = Node
        if (self.end == None):
            self.end = self.start

# Accepted
class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if (s == None or len(s) == 0):
            return 0
        result = 0
        k = 0
        subset = set()
        for i in range(len(s)):
            char = s[i]
            # if not in the subset, we add
            if char not in subset:
                subset.add(char)
                result = max(result, len(subset))
            else:
                # if in the subset, we use the slow runner 
                # to update
                while(k < i):
                    # if the elem at slow pointer equal, 
                    # we increment slow by one
                    if char == s[k]:
                        k += 1
                        break
                    else:
                        # if the elem not equal to
                        # the current character
                        # we remove the elem at slow pointer
                        # update until k is one step below i 
                        # or it is equal to the element at i
                        subset.remove(s[k])
                        k += 1
        return result

#Lyfx Hackerrank
#separate child
import copy
def minmoves(arr):
    if len(arr) == 0 or len(arr) == 1:
        return 0
    arr1 = copy.deepcopy(arr)
    arr2 = copy.deepcopy(arr)
    swaps1 = 0
    swaps2 = 0
    p1 = 0
    p2 = 0
    while(p2  < len(arr1)):
        if arr1[p1] != 0:
            if arr1[p2] == 0:
                temp = arr1[p2]
                arr1[p2] = arr1[p1]
                arr1[p1] = temp
                swaps1 += (p2 - p1)
                p1 += 1
        else:
            if arr2[p2] == 1:
                p1 = p2
        p2 += 1
    p1 = 0
    p2 = 0
    while(p2  < len(arr2)):
        if arr2[p1] != 1:
            if arr2[p2] == 1:
                temp = arr2[p2]
                arr2[p2] = arr2[p1]
                arr2[p1] = temp
                swaps2 += (p2 - p1)
                p1 += 1
        else:
            if arr2[p2] == 0:
                p1 = p2
        p2 += 1
    print(arr1, arr2)
    print(swaps1, swaps2)
    return min(swaps1, swaps2)

#Lyfx Hackerrank
# binary search

def binary_Search(arrary, n, x):
    low = 0
    high = n-1
    while(low <= high):
        mid = int((low+high)/2)
        if array[mid] <= x:
            low = mid + 1
        else:
            high = mid - 1
    return h
def counts(nums, maxes):
    nums1 = sorted(nums)
    countray = []
    for maxi in maxes:
        index = binary_Search(nums1, len(nums1), maxi)
        countray.append(index+1)
    return countray

#Accepted Answer
class MinStack:
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.min_elem = None
        self.stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if len(self.stack) == 0:
            self.min_elem = x
        else:
            self.min_elem = min(self.min_elem, x)
        self.stack.append(x)

    def pop(self):
        """
        :rtype: void
        """
        self.stack.pop()
        if len(self.stack) > 0:
            self.min_elem = min(self.stack)
        else:
            self.min_elem = None

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]
        

    def getMin(self):
        """
        :rtype: int
        """
        return self.min_elem
        


# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()

#Accepted
class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        first = m -1
        second = n -1
        total = m + n -1
        while (total >= 0):
            if (second < 0 or (nums1[first] > nums2[second] and first >= 0)) :
                nums1[total] = nums1[first]
                total -= 1
                first -= 1
            else:
                nums1[total] = nums2[second]
                total -= 1
                second -= 1

#Accepted Answer
class Solution:
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(s) == 0:
            return ""
        if len(s) ==1:
            return s
        
        longStr = s[0]
        for i in range(len(s)):
            temp = self.find_palin(s,i,i)
            if len(temp) > len(longStr):
                longStr = temp
            temp = self.find_palin(s,i,i+1)
            if len(temp) > len(longStr):
                longStr = temp
        return longStr
        
        
    def find_palin(self,s,first, second):
        while (first >= 0 and second <= len(s)-1 and s[first] == s[second]):
            first -= 1
            second += 1

        return s[first + 1 : second]
#Accepted Answer version 1
class Solution:
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        number = 0
        expo = 0
        for i in range(len(s)-1,-1,-1):
            number += (26**expo)*(ord(s[i]) - ord('A') + 1)
            expo += 1
        return number
#Accepted Answer version 2 forward
class Solution:
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        number = 0
        for i in range(len(s)):
            number = number * 26 + (ord(s[i]) - ord('A') + 1)
        return number

#Accepted Answer
class Solution:
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        match = {}
        match[")"] = "("
        match["}"] = "{"
        match["]"] = "["
        stack = []
        if len(s)== 0:
            return True
        if len(s) == 1:
            return False
        for letter in s:
            stack.append(letter)
            if letter in match.keys():
                if len(stack) == 1 or stack[-2] != match[letter]:
                    return False
                stack.pop()
                stack.pop()
        if len(stack) != 0:
            return False
        return True
#Accepted Answer
class Solution:
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        length_bool = [False]*(len(s) + 1)
        length_bool[0] = True
        
        for i in range(1,len(s)+1):
            for j in range(i-1,-1,-1):
                if length_bool[j] and s[j:i] in wordDict:
                    length_bool[i] = True
                    break
        return length_bool[len(s)]

#Accepted Answer
class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        result = ""
        for i in digits:
            result += str(i)
        result = int(result) + 1 
        final_str = [int(char) for char in str(result)]
        return final_str

#Accepted Answer: Version2
class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        for i in range(len(digits)):
            if digits[~i] < 9:
                digits[~i] += 1
                return digits
            digits[~i] = 0
        return [1] + [0]*len(digits)

#Answer
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        result = []
        tree_stack = []
        tree_stack.append(root)
        while (len(tree_stack) != 0 and root != None):
            temp = tree_stack.pop()
            result.append(temp.val)
            if temp.right != None:
                tree_stack.append(temp.right)
            if temp.left != None:
                tree_stack.append(temp.left)
        return result
#Accepted Answer
class Solution:
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        
        if numRows == 1 or len(s) == 1:
            return s
        inter = numRows - 2
        total_dimension = []
        iteration = 0
        last = numRows - 1
        time = 0
        if numRows == 2:
            for i in range(len(s)):
                if i < 2:
                    total_dimension.append(s[i])
                else:
                    total_dimension[(i)%2] += s[i]
            final_str = ""
            for i in total_dimension:
                final_str += i
            return final_str
            
        for i in range(len(s)):
            if iteration < numRows:
                if time == 0:
                    total_dimension.append(s[i])
                else: 
                    total_dimension[iteration] += s[i]
                iteration += 1
            elif iteration < numRows + inter:
                iteration += 1
                total_dimension[last-(iteration - numRows)] += s[i]
                if iteration == numRows + inter:
                    iteration = 0
                    time += 1
        final_str = ""
        for i in total_dimension:
            final_str += i
        return final_str

# Accepted Answer
class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        result_pre = ""
        end = 1
        if len(strs) == 1:
            return strs[0]
        if len(strs) ==0 or "" in strs:
            return result_pre
        violation = False
        while(True):
            temp = strs[0][0:end]
            for elem in strs:
                if end >  len(elem)  or temp != elem[0:end]:
                    violation = True
                    break
            if (violation):
                break
            end += 1 
            result_pre = temp
        return result_pre



#Accepted Answer
class Solution:
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        term = {}
        for i in range(1, n+1):
            if i == 1:
                t = "1"
            else:
                t = self.helper(term[i-1])
            term[i] = t
        return term[n]
        
    def helper(self,s):
        start = 0
        end = 1
        count = 1
        word = ""
        if len(s) == 1:
            return "1" + s
        while end < len(s):
            if s[start] == s[end]:
                count += 1
            else:
                word += (str(count) + s[start])
                start = end
                count = 1
            end += 1
            if end == len(s):
                word += (str(count) + s[start])
                
        return word

#Faster 

class Solution:
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        ans = "1"
        for i in range(1,n):
            temp = ""
            chars = list(ans)
            j = 0
            while (j < len(chars)):
                count = 1
                while ((j < (len(chars) - 1)) and (chars[j] == chars[j+1])):
                    count += 1
                    j+= 1
                temp += (str(count))
                temp += (chars[j])
                j+=1
            ans = temp
        return ans
        
#Accepted
class Solution:
    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        triangle = [[1]]
        if numRows == 0:
            return []
        if numRows == 1:
            return triangle
        for i in range(1, numRows):
            temp = []
            for j in range(i+1):
                temp.append(self.getElem(j-1,triangle[i-1])+self.getElem(j,triangle[i-1]))
            triangle.append(temp)
        return triangle
    def getElem(self,index,array):
        if index<0 or index == len(array):
            return 0
        else:
            return array[index]

# Accepted
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 ==[] and l2 == []:
            return []
        elif l1 == []:
            return l2
        elif l2 == []:
            return l1
        result = []
        temp = l1
        while(temp != None and l2 != None):
            if temp.val >= l2.val :
                cache = l2.next
                l2.next = temp
                temp = l2
                l2 = cache
            result.append(temp.val)
            temp = temp.next
        while(temp != None):
            result.append(temp.val)
            temp = temp.next
        while(l2 != None):
            result.append(l2.val)
            l2 = l2.next
        return result
# Fastest
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(-1)
        prev = dummy
        while l1 and l2:
            if l1.val <= l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 or l2
        return dummy.next

#Accepted
class Solution:
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        answer_stack = []
        special = [4,9]
        Translation_storage = {}
        Translation_storage[1] = "I"
        Translation_storage[5] = "V"
        Translation_storage[10] = "X"
        Translation_storage[50] = "L"
        Translation_storage[100] = "C"
        Translation_storage[500] = "D"
        Translation_storage[1000] = "M"
        tracker = 0
        modulus = num % 10
        while (num != 0):
            temp = ""
            if modulus in special:
                temp += Translation_storage[10**tracker]
                temp += Translation_storage[modulus*(10**tracker) + 10**tracker]
            else:
                if modulus >= 5:
                    temp += Translation_storage[5*(10**tracker)]
                    remainder = modulus - 5
                    if (remainder != 0):
                        for i in range(remainder):
                            temp += Translation_storage[1*(10**tracker)] 
                else:
                    for i in range(modulus):
                            temp += Translation_storage[1*(10**tracker)] 
            answer_stack.append(temp)
            num //= 10
            modulus = num % 10
            tracker += 1
        result = ""
        while (len(answer_stack) != 0):
            result += answer_stack.pop()
        return result

















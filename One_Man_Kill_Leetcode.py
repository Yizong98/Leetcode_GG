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